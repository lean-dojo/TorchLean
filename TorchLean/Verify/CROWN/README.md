# CROWN / LiRPA in TorchLean

The CROWN folder holds TorchLean's CROWN/LiRPA-style bound propagation: interval and affine passes
over graphs of registered operators, certificate data structures, and their proofs. The verify
workflows, external certificate checks, VNN-COMP suites and Lyapunov/controller experiments all run
on it. A computed box becomes a mathematical claim through the soundness theorems in `Op/`, each of
which holds on a successful real run of the graph. No pass promises a box for every graph.

## Main Files

The root module `TorchLean.Verify.CROWN`, the file `CROWN.lean` next to this folder, holds
interval boxes (`Box`) and the evaluation of an affine form `W x + b` on a box
(`TorchLean.Linear.evalOnBox`). It also holds `IBP.linear`, the interval rule for a linear layer
(endpoint products, no sign split), and the `IBP.matPos`/`IBP.matNeg` weight decomposition that
the CROWN linear rule (`propagateLinearBounds`) uses.

1. `FlatBox.lean` and `FlatBox/`: `FlatBox`, the interval container over flattened tensor values,
   and what the passes over both graph forms share and what depends on neither IR.
   `FlatBox/Transfer.lean` has the interval transfers on flat boxes, with their directed and real
   enclosure lemmas in `FlatBox/Directed.lean` and `FlatBox/Real.lean`. `Flatten.lean` flattens
   shaped values, `Matmul.lean` holds the interval matrix product, and `Concat.lean` holds
   concatenation as the concat layout reads it. `Conv.lean` is the interval convolution of weights
   in PyTorch's packed grouped layout, and `Affine.lean` and `Backward.lean` hold the affine bounds
   of flattened values and the directed backward coefficient state. `Affine/Real.lean` and
   `Backward/Real.lean` read those at real values for the α-CROWN and backward proofs.
   `Enclosure.lean` holds layout, dimension and enclosure lemmas. `Softplus.lean` bounds softplus
   over the reals, and `Refinement.lean` proves that subdividing a box loses no point.
   `Validity.lean` proves that over `ℝ` interval addition, subtraction, ReLU and the linear rules
   keep boxes valid (`lo ≤ hi` at every coordinate).
2. `BoundOps.lean` and `BoundOps/`: the endpoint interfaces `BoundOps` and `NonlinearBoundOps`
   with the host `Float` and `Float32` instances, and their real-number laws
   (`BoundOps/Lawful.lean`). `BoundOps/Interval.lean` holds real interval arithmetic and
   interpreted directed-endpoint laws, including the four-corner product bound reused from
   FloatLib, the exact real `LawfulNonlinearBoundOps` instance, and `BoundOps.outward` with its
   lawful and nonlinear companions, which round exact real results outward with a FloatLib
   rounder. `BoundOps/FP32.lean` and `BoundOps/Binary32.lean` hold the binary32 instances of the
   proof model and of FloatLib's executable arithmetic.
3. `MLP.lean` and `MLP/`: interval bounds for a two-layer ReLU MLP on vectors (`TwoLayerMLP`,
   `TwoLayerMLP.boundIBP`). `TwoLayerMLP.boundIBP_sound` proves the interval bound sound over `ℝ`.
   `MLP/Distillation.lean` holds the real-valued IBP check that a student two-layer MLP stays
   within `eps` of its teacher on an input box, with its soundness theorem.
4. `Graph.lean`: the flat affine bounds of the CROWN passes (`FlatAffineBounds`) and the uniform
   `ℓ∞` input box around a tensor (`FlatBox.lInfBall`).
5. `Conv.lean` and `Conv/`: CROWN-IBP bounds and an exact flattened affine map for
   arbitrary-dimensional convolution, with the convolution matrix proof in `Conv/Affine.lean` and
   the real box facts of the enclosure proof in `Conv/Enclosure.lean`.
6. `LayerNorm/`: directed LayerNorm arithmetic (`LayerNorm/Directed.lean`), the LayerNorm value
   enclosure (`LayerNorm/Enclosure.lean`), and the rowwise LayerNorm derivative bounds
   (`LayerNorm/Derivatives.lean`) with their enclosure proof
   (`LayerNorm/DerivativeEnclosure.lean`).
7. `Activation.lean` and `Activation/`: the canonical executable ReLU relaxations and elementwise
   interval maps used by the CROWN passes, kept apart from heavier proof imports.
   `Activation/ReLU.lean` proves the scalar ReLU upper relaxation, and `Activation/IBP.lean` the
   monotonicity and Lipschitz facts behind the nonlinear IBP transfers.
8. `Op.lean` and `Op/`: the passes over graphs of registered operators (interval propagation,
   backward CROWN and forward α-CROWN), the IBP certificate, and their soundness theorems, listed
   file by file under `Op/` Modules.
9. `Arithmetic.lean`: `withArithmetic`, which runs a bound computation under native `Float32` or
   the bit-level `ExecFloat.Binary 8 23` and rejects complex arithmetic.
10. `Widget.lean`: the infoview panels `#crown_view` and `#bounds_tightness_view`.

## Bound Computation And Proofs

CROWN-style code has three distinct jobs:

- represent bounds and affine relaxations,
- compute or check transfer rules over supported operators,
- prove that accepted bounds imply a semantic property of the graph or model.

Theorem coverage differs between passes. The proof files name the fragments with Lean support,
and outside them the executable workflows and JSON checkers serve as diagnostics or artifact
checks. When writing a claim, cite the strongest support available: a runtime report, a checked
certificate, a transfer-rule assumption or a theorem.

Support depends on the IR operation, tensor geometry, parameter payload and scalar backend, and a
model that builds or lowers may get no box from a pass. The [verification
guide](../../../home_page/blueprint/TorchLeanBlueprint/Guide/Ch4_Verification/Verification.lean)
states the theorem fragments and their premises. Runtime support is broader.

## Claim Shapes

The same graph supports several levels of claim, and a claim should say which one it makes.

| Claim | Evidence to cite |
| --- | --- |
| A bound pass ran | the runtime command, graph and output ids, input box and printed bound |
| An artifact was accepted | the checker module, schema, artifact path and recomputed predicate |
| A certificate is sound | the `Op/` soundness theorem, graph semantics and checked hypotheses |
| An external verifier found the leaf | the external producer and the Lean-checked leaf artifact |
| A finite-precision bound is used | the scalar format and the caller's numerical correspondence |

For the checked exact-real result on finite binary32 artifacts, see
[certificate acceptance](#certificate-acceptance) below.

An alpha-beta-CROWN leaf artifact, for example, holds a root box and the exported terminal leaves
of a branch-and-bound run, each with its box, lower bounds, thresholds and an optional witness
index and margin. A claim about the network also needs provenance for the external run that
produced the leaves.

## Bound Passes And Proofs

The passes run on graphs of registered operators (`IR.Graph`). A lowered model reaches them
as an import (`IR.LoweredModel.toImported`, `IR.Imported`), and a loaded artifact of format
`torchlean.ir` as the graph of its program (`Program.graph?`, with the payload `Payload.graph`) when
every step is an operator node at one scalar. A typical workflow builds or imports a graph,
attaches an input box, computes the IBP box of every node and then propagates affine forms for a
chosen output or objective. `Op/Exec.lean` and `Op/ExecCROWN.lean` hold the transfer rules, and the
proof files say which rules have soundness theorems and under which assumptions.

### Tensor geometry

The graph stores tensor shapes, while its boxes and affine coefficients use flat coordinates. The
shape decides which coordinates take part in each operation:

| Operation | Executable transfer contract |
| --- | --- |
| `cat dim` | Any valid dimension, at least one parent, and matching dimensions off `dim`. |
| Binary matmul | Broadcastable leading axes, vector promotion, and matching inner dimensions. |
| Convolution | Checked groups, dilation, stride, asymmetric padding, and leading batch axes. |
| `reduce .sum`, `mean` | A sum over all dimensions, or a sum or mean over one nonempty dimension. |
| Average pooling | Any pooled axes, kernels, strides, padding, dilation and ceil mode. |
| `batchNorm dim` | Running statistics, a valid channel dimension, and matching parameter lengths. |
| `layerNorm` | The last axis only, scale and bias of shape `[embedDim]`, and a nonempty lead. |
| `softmax dim` value range | Any valid dimension without a mask, the last one with a hard mask. |
| Elementwise interval maps | Preserve arbitrary shapes, including scalar and empty shapes. |

The sum and mean transfers reject `keepdim`. Over a single dimension they keep the others, including
empty leading ones. Sums use directed accumulation, and means and average pooling also enclose the
count before dividing. Average pooling divides by the number of window cells in the padded input
under `countIncludePad` and in the input otherwise. BatchNorm keeps the input shape and propagates
intervals through stabilization, square root, normalization, scale and bias. It returns no bound
when it can't establish finite arithmetic or a positive denominator. Unmasked softmax gives `[1,1]`
when the dimension has extent one and `[0,1]` otherwise. `Op.Directed.encloses_of_run` covers each
of these transfers.

Concatenation keeps parent order, including empty dimensions and repeated parent ids. Values,
interval endpoints and affine rows follow the same coordinate map. Backward propagation splits the
output coefficients along that map and accumulates every occurrence of a repeated parent, so for
`cat(x, x)` both coefficient slices contribute to `x`. The shared interval rule (`Op.shared?`)
places its inputs' boxes by the same layout (`Op.catLayout`), and `Op.flat_cat`
(`Op/RunValues.lean`) proves that a concatenation's flat value is the flat concatenation of its
inputs' flat values over that layout.

For matrix multiplication, the leading axes broadcast before the final two axes contract.
For example, `[1,4,5]` times `[2,5,6]` produces `[2,4,6]`. A vector is promoted for the
contraction and its introduced axis is removed afterward: `[5]` times `[2,5,6]` produces
`[2,6]`, while `[5]` times `[5]` produces a scalar.
The backward pass bounds a binary product by McCormick planes of its operand boxes where the
coefficient has one sign and by their interval product otherwise, so this relaxation can lose
correlations.

`layerNorm` normalizes the last axis: on `[2,2,2]` it normalizes four rows of two coordinates,
with scale and bias of shape `[2]`. The leading axes must hold at least one row and the last axis
at least one coordinate. The elementwise maps in `CROWN.IBP`
(`mapMinmax`, `sigmoidMinmax`, `tanhMinmax`, `sinMidpoint` and `cosMidpoint`) apply the same
scalar formula at every coordinate and keep the input shape.

A convolution node (`Op.conv`) in PyTorch's layout reads its kernel as `(outC, inC / groups, k...)`,
which keeps for each output channel the input channels of its group (`Conv.packedKernel`).
Its interval box (`Op.convOpBox?`) runs the flat-box transfer on the dense layout
`(outC, inC, k...)` that `denseKernel` builds from it, with zeros outside the groups, and is none
for another layout. The backward pass sends coefficients through a convolution by the transposed
convolution over the same configuration (`CROWN.Op.convBack?_sound`), so it builds no dense matrix.
General shape support is not a memory or speed guarantee.

In `Conv/Affine.lean`, `CROWN.convLinearMatrix_spec` proves that the
dense matrix of a convolution, with one row per output scalar and one column per input scalar,
applied to the flattened input, plus the broadcast bias, equals the flattened grouped
convolution over real tensors. It covers dilation, asymmetric padding and leading batches.
Graph execution also validates the configuration, and rounded execution needs its own numerical
correspondence.

`Conv/Derivatives.lean` differentiates the grouped convolution in its input over any leading
batch shape and proves its input adjoint.

`FlatBox/Conv.lean` proves enclosure for the directed convolution's channel and kernel
folds over real endpoints, and relates their result to the same grouped convolution. On graphs of
registered operators, `Op.Directed.encloses_conv` proves that the convolution box encloses the
real convolution at the reals of the rounded kernel and bias, at every lawful scalar and so at
`ℝ`, through `Op.convBox?_of_convOpBox?` (`Op/Conv.lean`). Convolution is part of the
certificate's soundness (`Op.certLocalOK_encloses`) and of `Op.Directed.encloses_of_run`.

`LayerNorm/Enclosure.lean` proves that LayerNorm's directed row sequence, including the
directed sum and count of the mean, encloses `Tensor.layerNorm` under the endpoint and nonlinear
operation laws, with zero and one interpreted exactly.

`BoundOps` holds the executable lower and upper operations, and `LawfulBoundOps` reads their
endpoints as real numbers and proves each lower result below the exact operation and each upper
result above it. An enclosure lemma needs both, since executable operations alone prove nothing.
`NonlinearBoundOps` holds interval transfers for division, square root, transcendental functions,
bounded activations and layer normalization. A backend returns `none` when it has no justified
enclosure, and the graph pass then leaves that node unresolved and puts no host-library value in
its place. `LawfulNonlinearBoundOps` proves that every successful nonlinear transfer encloses the
real operation and that a returned layer-normalization bound `layerNormAbsBound n` is at least
`Real.sqrt n`.

`LawfulBoundOps` also interprets the context's `max` and `min`, and its IEEE `Context.maximum` and
`Context.minimum`, as the real maximum and minimum. The interval rules select endpoints with
`Context.minimum` and `Context.maximum`, which keep a NaN, and the graph pass gives a node no box
when an endpoint of its box is a NaN.

The instances differ in what they can bound:

- `ℝ`: exact arithmetic, with `LawfulBoundOps` holding by definition (`instLawfulBoundOpsReal`).
- `ℚ`: the scoped exact instance and its laws in `TorchLean.Verify.Certificate.Rational`
  (`instLawfulBoundOpsRat`).
- `FP32` (`BoundOps/FP32.lean`): exact real results rounded outward to the binary32 grid
  (`BoundOps.outward` at the binary32 rounder), with proved `LawfulBoundOps` and
  `LawfulNonlinearBoundOps` instances. It is a noncomputable proof model with an unbounded
  exponent, so it has no overflow, infinities or NaNs.
- FloatLib binary32 (`ExecFloat.Binary 8 23`, `BoundOps/Binary32.lean`): endpoint
  operations from FloatLib's arithmetic, rounded down and up. Division, `exp`, `log`, square
  root, sine and cosine get FloatLib's rational enclosure of the whole image, rounded outward to
  binary32, and sigmoid and tanh get their codomain bounds. A reversed box, NaN, infinity,
  division through zero, `log` or square root outside its domain, and overflow give `none`. The
  `CROWN.Binary32` theorems (`divBounds_sound` to `cosBounds_sound`) prove, on exactly decoded
  finite endpoints, that every successful result contains the real image of its input interval.
  The carrier holds NaNs and infinities, so it has no global `LawfulBoundOps` instance.
- host `Float` and `Float32`: each basic operation, division of a finite interval by a finite
  interval that avoids zero and square root widened by one adjacent value, and `none` for `exp` and
  `log`. `Float` gives sigmoid, tanh, sine and cosine their codomain bounds, and `Float32` gives
  sine and cosine theirs. Neither has a `LawfulBoundOps` instance.

`Op.runIBP` records an unavailable transfer as `none`, and an output query fails when its requested
box is missing. The `native` CLI backend runs at `Float32`, which has no box for `exp` or `log`.
The `ieee` backend runs at FloatLib binary32 and bounds both, apart from the `none` cases above.
Softmax can use the coarse coordinate ranges of the geometry table on either backend. The `native`
backend also leaves sigmoid and tanh unresolved, and the `ieee` backend bounds sigmoid by `[0, 1]`
and tanh by `[-1, 1]`, their global codomain bounds.

`Op.outputBoxCROWN?` runs the backward pass on every backend. At a node without a coefficient
rule, such as ReLU, layer norm, softmax or MSE loss, the pass bounds its active objective against
the node's IBP box and propagates no coefficients to the parents. Binary products take their
McCormick planes. A CROWN result need not tighten IBP. The soundness theorems of the passes are in
`Op/`: `Op.Directed.encloses_of_run` for interval propagation at any scalar with lawful
endpoint operations, `CROWN.Op.outputBoxCROWN?_of_run` and its siblings for the backward
queries, and `Op.Alpha.encloses_of_run` for the forward α-CROWN pass at `ℝ`. Each holds on a
successful real run of the graph.

The backward sweep uses no per-neuron ReLU slopes, since the α slopes belong to the forward pass.
Relating a rounded result to a separate native execution also needs the finite-precision bridge
of [`TorchLean/Verify/RuntimeApprox`](../RuntimeApprox/README.md).

Use this split when adding operators:

1. Add the executable interval/affine transfer rule.
2. State the shape and payload assumptions it needs.
3. Add or extend the proof layer soundness theorem when the operator supports formal
   graph-certificate claims.
4. Add a small verifier example or fixture if the rule is exposed through `lake exe verify`.

### Certificate acceptance

The library functions `IBPNode.check` and `CROWNNode.check` check node certificates, and
`lake exe verify` has no command for them. A CROWN certificate may carry ReLU phases, which only
fix ReLUs whose IBP interval already proves the phase. The CROWN check ends with a pure complete
replay, and `CROWNNode.accepts_sound` turns a successful binary32 replay into local
consistency (one entry per node, each its replay with the operator graph's forward pass
`Op.alphaStep?`). The result is a theorem about the imported binary32 transcript, which need not
enclose the graph, because the replay rounds to nearest. After the layers `h = x + 1` and
`y = h + 2 ^ -24` at `x = -1`, where `y = 2 ^ -24`, the replayed upper form of `y` is `x + 1`,
which is `0` there. On `y = relu x` over `[-M, M]`, with `M` the largest finite value,
`M - (-M)` overflows and the replayed upper form of `y` is `0`, below `relu M = M`.

For nonempty vector linear/ReLU chains, `Certificate.FiniteArtifact.accepts` checks the
supplied bounds against exact rational transfers, with no binary32 replay. It rejects inward-rounded
affine coefficients even when they agree with binary32 replay, and accepts bounds from any producer
that dominate the exact transfers. `FiniteArtifact.accepts_graph_sound` (in
`TorchLean.Verify.Certificate.FiniteArtifact.Semantics`) then proves that the original graph, read
with the exact real values of those binary32 parameters, satisfies every requested output inequality
on the whole input box. The checker needs bounds for the whole chain and at least one inequality.
Native execution error is a separate obligation.

## `Op/` Modules

- `Op.lean`, `Op/RunValues.lean`: node and constant access (`Op.constAt?`), the port layouts of a
  matrix product and a concatenation (`Op.matmulLayout`, `Op.catLayout`) and topological order
  (`Op.TopoSorted`). At `ℝ`, `Op/RunValues.lean` holds the facts about a successful run that the
  enclosure proofs read (`Op.runVals` and the flat values of products and concatenations).
- `Op/Imported.lean`: the passes on the operator graph of an import (`IR.Imported`), which also
  holds the payload, from a box at graph input `0`. `Imported.runIBP` returns the box of every node,
  and `Imported.outputBoxCROWN?` gives the CROWN box of the first output.
- `Op/Cert.lean`, `Op/Certify.lean`, `Op/CertSoundness.lean`: the IBP certificate on graphs of
  registered operators, with rules keyed on the operator and parameters as constant nodes.
  `Op.certify` computes it. Every box of a table that is the certificate step's fixpoint at every
  node (`Op.CertLocalOK`) encloses its node's flat value in a table of node outputs that is the
  fixpoint of the node evaluations (`Op.SemLocalOK`). Those tables are the successful real runs
  (`Op.semLocalOK_iff_run`). The enclosure theorem `Op.certLocalOK_encloses` assumes that the seed
  boxes enclose the graph inputs and has no premise on the operators, shapes or node order.
  `IBPNode.check` compares the boxes of a node certificate with those of `Op.runIBP` at binary32,
  not with `Op.cert?`.
  `Op/Conv.lean` relates a convolution's flat value and box to the grouped convolution
  (`Op.flat_conv`, `Op.convBox?_of_convOpBox?`).
- `Op/Exec.lean`: interval bound propagation on graphs of registered operators at any scalar with
  directed endpoint operations (`Op.runIBP`). An operator gets the shared rule `Op.shared?` when
  it has one and a directed rule of its own otherwise (`Op.directed?`), which for tanh, sigmoid,
  sine and cosine is the backend's enclosure. The verify workflows `torchlean-ibp`,
  `torchlean-crown-ops` and `torchlean-transformer-ibp` run their IBP on it. `Op/ExecCert.lean`
  relates it to the certificate at `ℝ`. On a graph without those four operators
  (`NoBackendEnclosure`, no operator of kind `backend` in `Op.ruleKind`) the engine has every box
  of the certificate (`Op.runIBP_of_certify`). Those boxes enclose the node's value in every
  successful real run whose inputs the seeds enclose (`Op.runIBP_encloses_of_run`). On a graph
  whose operators all have kind `shared` or `none` (`SharedOnly`) the engine's table is the
  certificate (`Op.runIBP_eq_certify`), and on a sorted such graph it satisfies `Op.CertLocalOK`
  (`Op.runIBP_localOK`).
- `Op/Linear.lean`: over `ℝ` the linear-layer rule `Op.ibpLinearAt` keeps a valid input box valid
  (`Op.ibpLinearAt_valid_real`), and at any scalar it gives a box of the layer's output width
  (`Op.ibpLinearAt_dim`). It boxes one row: at a nonempty extent, an input with more than one
  leading row gets no box.
- `Op/DirectedBasic.lean`, `Op/DirectedNonlinear.lean`, `Op/Directed.lean`: `Op.runIBP` at a scalar
  with lawful endpoint operations. The table `Op.ruleKind` gives each operator the kind of its
  rule, and every operator the engine boxes has a kind other than `none` (`Op.Directed.Supported`,
  `Op.Directed.supported_of_ibp?`). These are copies, pointwise arithmetic, negation, squaring,
  absolute values, elementwise maxima and minima, ReLU, linear layers, plain and scaled matrix
  products, and the backend enclosures of exp, log, sqrt, reciprocal, tanh, sigmoid, SiLU, sine and
  cosine. They also include softplus, `safeLog` with its scalar epsilon, sums over every axis in
  order or over one axis and means over one axis, without `keepdim`, sum, average, maximum and
  minimum pooling, and batch norm with running statistics. The list continues with layer norm,
  scaling, leaky ReLU, ELU and clamping by scalar constants, GELU, sinh, cosh, log-softmax,
  cumulative sums, select, padding, gather, embedding, sign, floor, ceiling, rounding, truncation
  and fractional part. It goes on with arctangent, arcsine, arccosine, erf, `exp x - 1`, `2 ^ x`,
  `arsinh`, tangent, `log (1 + x)`, `log2`, `log10`, the reciprocal square root, `fmax`, `fmin`,
  true and truncating division, atan2 and hypot. Contraction, fmod, remainder, copysign, softmax
  along an axis, masked softmax along the last axis, uniform and Bernoulli mask samples and mean
  squared error are boxed too. So are the re-indexings permute, slice, flip, broadcast and
  `indexAt`, concatenation along an axis, convolution, and maxima and minima over any valid axes
  with either `keepdim`. Every box the engine returns encloses its node's flat value in a
  successful real run whose payload holds the reals of the rounded constants and whose graph
  inputs the seed boxes enclose (`Op.Directed.encloses_of_run`). `CROWN.Op.ibp?_encloses` proves
  the rule of one node sound on any real values its references read (`Op.Directed.RuleEncloses`),
  and `Op.Directed.step?_encloses` gives the step of one node of any sound rule on any table of
  boxes.
  The operators' enclosure lemmas are in `Op/DirectedBasic.lean` and, where they need the nonlinear
  endpoint operations, in `Op/DirectedNonlinear.lean`. A layer norm's box is the directed box of
  each row met with the uniform range of each entry, and it encloses the real layer norm
  (`Op.Directed.layerNormBox?_encloses`, `Op.Directed.layerNormRange_encloses`).
- `Op/Alpha.lean`: the forward pass (`Op.alphaStep?`) at `ℝ`. Every affine bound it gives a node
  encloses, at the input point, the node's flat value in a successful real run
  (`Op.Alpha.encloses_of_run`). The theorem assumes that the interval table encloses the run, that
  the designated input's value is the point and that the slopes lie in `[0, 1]`.
  `CROWN.Op.alpha?_encloses` is the rule of one node, and `CROWN.Op.runAlphaCROWN_encloses`
  covers the engine's own table. `FlatBox/Affine/Real.lean` holds the real relaxation lemmas the
  alpha-CROWN and phased alpha-CROWN proofs share (the sign-split linear rule, the ReLU and phase
  relaxations) and the reading of affine bounds at an input point (`EnclosesAtInput`).
- `Op/ExecCROWN.lean`: CROWN on graphs of registered operators from an interval table. The backward
  pass (`Op.backwardObjective?`, `Op.affineBounds?`, `Op.nodeBounds?`, `Op.outputBoxCROWN?`,
  `Op.backwardObjectiveBox?`) runs a lower and an upper sweep of interval coefficients from an
  objective to the input. Products of two bounded operands (`mul`, `matmul`, `scaledMatmul`) take
  McCormick planes, and a convolution with constant kernel and bias takes the transposed
  convolution. Each direction keeps the form of the shared rules alone when that is tighter on the
  input's box. `Op.affineBounds?` takes a matrix objective at a node. `Op.crownBoxes` is full CROWN
  on it (each node's box within its affine bounds on the input box), and `Op.runCROWN` reads every
  node's affine bounds over those boxes, per row the tighter of the query's and the box's constant
  forms. Every box a wrapper returns is met with its node's interval box (`Op.meetWith`), and an
  objective's box with the objective evaluated on that box. There is no whole-graph check. On an
  imported graph each node refuses for itself, a box wrapper without the node's interval box is an
  error, and a sweep that cannot pass a node leaves the node's interval box. The forward pass
  (`Op.alphaStep?`, `Op.runAlphaCROWN`, `Op.outputBoxAlphaCROWN?`) gives α-CROWN affine bounds with
  ReLU slopes and phases, and the verification API's `alphaCROWN` algorithm runs it on the graph of
  an import. At a scalar without exact affine reassociation, every rounded one included,
  `Op.outputBoxAlphaCROWN?` returns the box of `Op.outputBoxCROWN?`.
- `Op/Backward.lean`, `Op/BackwardProduct.lean`: the backward pass of `Op/ExecCROWN.lean` at a
  scalar with lawful endpoint operations. The theorems hold at a supplied
  real point of the graph (`CROWN.Op.GraphPoint`) at which every operator node satisfies its real
  equations (`CROWN.Op.Equation`, `CROWN.Op.ProductEquation`), whether or not a query
  reaches the node. The interval table encloses a `GraphPoint`, and at it every constant node whose
  rounded value reads and flattens holds the reals of that flat value. The shared rules are sound
  (`CROWN.Op.directedRule_sound`, and at a convolution the transposed convolution rounded outward
  encloses the real adjoint, `CROWN.Op.convBack?_sound`), so a sweep of them bounds the objective
  at the input's coordinates from below and above (`CROWN.Op.sweep_encloses`). The McCormick
  product rules are sound in their direction (`CROWN.Op.backwardRule_sound`), so a product sweep of
  direction `dir` bounds the objective in that direction (`CROWN.Op.sweep_bound`). The queries
  `Op.backwardObjective?`, `Op.affineBounds?`, `Op.nodeBounds?`, `Op.backwardObjectiveBox?` and
  `Op.outputBoxCROWN?` enclose what they bound (`CROWN.Op.outputBoxCROWN?_encloses` and its
  siblings). `FlatBox/Backward/Real.lean` holds the lemmas they use about the coefficient state
  and affine forms: the real reading of the state, the transposed linear, slice, broadcast and
  concatenation rules, and the reverse fold over the positions.
- `Op/BackwardRun.lean`: the point and the equations from a successful real run of the graph on a
  payload holding the reals of the rounded constants (`CROWN.Op.graphPoint_of_run`,
  `CROWN.Op.equation_of_run`, `CROWN.Op.productEquation_of_run`). With them the queries
  enclose the run's flat values with no premise on the point (`CROWN.Op.outputBoxCROWN?_of_run`
  and its siblings, and `CROWN.Op.runIBP_encloses`). So do the boxes and affine bounds of full
  CROWN (`CROWN.Op.crownBoxes_encloses`, `CROWN.Op.runCROWN_encloses`).

## Optional input subdivision

`Graph.Refinement.bound enclose input budget` (`FlatBox/Transfer.lean`) refines the result of an
enclosure procedure by splitting its input box. With interval bound propagation it uses the
transfers of every supported layer, with no Transformer-specific path. Other graph
inputs and parameters keep their original boxes and values.

For example, over `x ∈ [-1, 1]`, independent interval evaluation of
`ReLU(x) + ReLU(-x)` gives approximately `[0, 2]`. Splitting at zero and taking the hull of both
results gives approximately `[0, 1]`. Likewise, splitting can improve the lower bound on `x * x`.
A dense layer with independent input coordinates may already have tight bounds and see no gain.
Input-independent activation fallbacks such as `[0, 1]` also need not improve.

The budget counts binary splits over the whole split tree, so there are at most
`2 * budget + 1` IBP calls. Zero budget returns the ordinary enclosure. No representable interior
midpoint means no split. Both child results are required, and a failed branch keeps the parent
result. Child results are combined by a hull and then intersected with the parent bound, so a
successful refinement never widens it. Invalid boxes or incompatible output dimensions are
rejected before they can become graph output certificates.

The artifact checker accepts the same option:
`Certificate.IBP.check G P inputs output path (refinement := some (k, budget))`,
splitting the box of graph input `k`. Its default performs a single pass, so a split budget is a
runtime and precision tradeoff the caller opts into.

In `FlatBox/Refinement.lean`, `Graph.Refinement.splitAt_covers` proves that every real input in a
parent box belongs to at least one child. The enclosure procedure also needs sound transfer rules:
subdivision proves nothing about rounded LayerNorm or other backend operations.

## Infoview Panels

The panels of `Widget.lean` show CROWN and IBP bound propagation over a graph of registered
operators (`IR.Graph`): which nodes got bounds, the shapes and flattened dimensions the
propagation works on, and small previews of the vectors and matrices involved. The panels are for
debugging and teaching: rendering a bound does not check it. They build on the tensor panel of
[`TorchLean/NN/Widget`](../../NN/Widget/README.md).

`#crown_view g, boxes, affine` shows a per-node table of the interval boxes of `CROWN.Op.runIBP`
and the affine bounds of `CROWN.Op.runCROWN` (`#crown_view g, boxes` without affine bounds), and
`#bounds_tightness_view g, boxes` shows the interval width `hi - lo` per node.

Import `TorchLean.Verify.CROWN.Widget`. The [widget
gallery](../../../TorchLeanExamples/DeepDives/WidgetGallery.lean) runs both commands on a small
graph.

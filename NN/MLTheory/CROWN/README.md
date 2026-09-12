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
   and `GraphRuntimeBridge.lean` relates the runtime evaluator to the semantics (`evalNode_bridge`).
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

## Claim Shapes

The same graph can support several levels of claim, and the wording should identify which one is
being used.

| Claim | Evidence to cite |
| --- | --- |
| A bound pass ran on a graph | the runtime command, graph id/output id, input box, and printed bound result |
| A JSON artifact was accepted | the checker module, schema name, artifact path, and recomputed predicate |
| A graph certificate is sound | the theorem in `Proofs/`, the graph semantics, and the hypotheses discharged by the checker |
| An external verifier found the leaf | the external producer/provenance plus the Lean-checked leaf artifact |
| A finite-precision bound is being used | the `FP32`, `IEEE32Exec`, or runtime bridge assumptions named by the caller |

For example, an alpha-beta-CROWN leaf artifact represents one exported terminal leaf: boxes, lower
bounds, thresholds, labels, and the witness comparison represented by the schema. A full producer
claim additionally needs provenance for the external branch-and-bound run that generated the leaf.

## Graph Engine And Proofs

The graph engine works over `NN.IR.Graph` node ids and payload stores. A typical workflow creates or
imports a graph, attaches an input box, computes per-node IBP boxes, and then propagates affine
forms for a selected output or objective. Operator files provide the transfer rules; proof files say
which rules have soundness statements or which assumptions remain.

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
- `IEEE32Exec` uses proved directed binary32 division and square root, while exponential and
  logarithmic transfers remain unavailable. Finite-path soundness is stated in the IEEE semantics
  modules rather than as a global ordered instance over NaNs and infinities;
- host `Float` widens basic binary64 operations by one adjacent value and does not claim directed
  transcendental-library results or provide a global `LawfulBoundOps` instance.

Forward affine CROWN uses a constant affine form when only an IBP enclosure is justified. This is
less precise than an analytic relaxation but preserves the checked interval. Objective-dependent
backward CROWN performs algebraic coefficient propagation; a result over an executable floating
type is not, by itself, a theorem about rounded runtime execution. Such a claim still needs the
finite-precision bridge described in `NN/Proofs/RuntimeApprox`.

Use this split when adding operators:

1. Add the executable interval/affine transfer rule.
2. State the shape and payload assumptions it needs.
3. Add or extend the proof layer soundness theorem when the operator supports formal
   graph-certificate claims.
4. Add a small verifier example or fixture if the rule is exposed through `lake exe verify`.

That keeps runtime diagnostics, accepted certificates, and theorem-backed graph claims connected
while preserving the distinction between execution evidence, checker acceptance, and theorem-backed
graph claims.

For node certificates, the executable α-CROWN and α/β-CROWN commands finish with a pure complete
replay. `certificateAccepts_eq_true` and `AlphaBetaCROWNNodeCertificate.accepts_eq_true` turn a
successful binary32 replay into `CrownCertLocalOK`. The result is a theorem about the imported
`IEEE32Exec` transcript. Applying a real-semantic enclosure theorem additionally requires the
appropriate transfer and finite-precision refinement hypotheses.

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
Its default remains the original single pass. A split budget is an explicit runtime/precision
tradeoff, rather than a hidden cost added to every verification request.

In `NN.MLTheory.CROWN.Proofs.GraphRefinement`, `Graph.Refinement.splitAt_covers` proves that every real input in a parent box belongs to at least
one child. The enclosure procedure still needs sound transfer rules: subdivision does not prove
universal soundness of rounded LayerNorm or other backend operations. The maintained tests cover
containment, actual tightening, Float/Float32/IEEE32Exec, multiple inputs, failed branches, invalid
endpoints, and the artifact-checker option.

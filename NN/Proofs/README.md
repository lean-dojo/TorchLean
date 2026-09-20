# `NN/Proofs`

This directory contains TorchLean's proof library. The files here are where runtime objects,
specifications, graph semantics, and verification checkers begin to acquire theorem-level meaning.

The curated import is:

```lean
import NN.Proofs
```

Ordinary model code should not import this directory. Use it when the file is proving a fact about
the spec layer, the runtime approximation layer, a graph translation, a floating-point model, or a
verification checker.

## What The Proofs Connect

TorchLean keeps several objects separate:

- spec-layer definitions in `NN/Spec`,
- executable runtime paths in `NN/Runtime`,
- graph semantics in `NN/IR`,
- certificate and artifact checkers in `NN/Verification`,
- theory-facing objects in `NN/MLTheory`.

The proof library connects those layers with explicit hypotheses. It should be possible to tell
from a theorem whether it is about exact real arithmetic, executable IEEE-style semantics, a
runtime approximation relation, an imported certificate, or a trusted native boundary.

## Folder Map

| Folder | What it proves |
| --- | --- |
| `Tensor/` | Tensor algebra, folds, bounds/norms, finite linear algebra, and factorization facts. |
| `Autograd/` | Selected reverse-mode/autograd correctness facts, Fréchet derivative rules, tape algebra, runtime links, and training-step algebra. |
| `RuntimeApprox/` | Approximation relations between spec-level operations, graph forward/backward computations, normal-form operator rules, convolution, softmax-axis, FP32/CROWN bridges, and scale/tolerance lemmas. |
| `Models/Attention/` | Attention invariants: causal masks, weights, and permutation/equivariance properties. |
| `Analysis/` | Analytic facts for softmax, normalization, dropout, FFT, Lipschitz-style statements, and related helper theory. |
| `Gradients/` | Smaller gradient facts for layers and activations. |
| `RL/` | MDP, environment, replay-buffer, Gymnasium-boundary, DQN/PPO-adjacent, and checked-runtime RL facts. |
| `Verification/ODE/` | ODE enclosure and corridor facts used by learned sub/supersolution checkers. |
| `Probability/` | Probability and diffusion-forward helper facts. |
| `Utils/` | List and real-function helper lemmas shared by the other folders. |

## Autograd Proofs

Autograd proofs in TorchLean are stated at the level where the mathematics is stable enough to
reuse. A theorem names the tensor expression, tape fragment, runtime link, or training-step update
that it covers, and the surrounding examples can then point to that theorem instead of relying on a
successful run as evidence. The core pieces are:

- algebraic correctness for selected tensor expressions,
- Fréchet derivative rules for elementwise functions, softmax, log-softmax, and MLP/MSE fragments,
- tape-level soundness and runtime-link lemmas for supported fragments,
- training-step algebra that lets optimizer and gradient facts be stated without hiding state
  updates in an opaque callback.

Start with `Autograd/Overview.lean` when navigating this area. The runtime link is under
`Autograd/Runtime/Link/`. `Core.lean` lowers the proof-layer `Graph` to the executable
`Runtime.Autograd.Tape`; `BackwardDense.lean` shows that the executed sweep
`Tape.backwardDenseAll` agrees with the proved sweep `Tape.backwardDenseFrom` on every
`ZeroPreserving` tape (`backwardDenseAll_eq_backwardDenseFrom`); `BackwardDenseGraph.lean`
discharges that hypothesis for lowered tapes (`lowerGraphToTape_zeroPreserving`) and states the
corollaries `backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx` and, over `ℝ`,
`backwardDenseAll_lowerGraphToTape_adjoint_fderiv`: the executed backward pass returns the adjoint
of the Fréchet derivative of the forward map. The per-node case analysis is shared through
`BackwardLeaves.lean` and `BackwardSnoc.lean`. These are statements about the exact tape model
at the given carrier and say nothing about `Float` rounding or CUDA.

Analytic derivative facts sit under `Autograd/FDeriv/` (`SoftmaxSpec.lean` gives
`hasFDerivAt_softmaxSpec_vec`, `softmaxFDerivCorrect`, `softmaxBackwardSpec_eq_vjp`, and the
log-softmax analogues) and `Autograd/Tape/Ops/` (attention:
`backpropVec_eq_adjoint_fderiv_scaledDotProductAttention`; normalization:
`layerNormJvp_layerNormBackward_adjoint`, `hasFDerivAt_batchNorm`,
`fderiv_batchNorm_eq_batchNormJvp`; the LayerNorm theorems assume `0 < ε`).

## Runtime Approximation

The runtime approximation files are the bridge between idealized math and executable runs. They
define relations such as "this runtime graph output stays within this tolerance of the spec output"
and then prove local operator rules.

This is the right place for facts that are weaker than exact equality but stronger than a test:

- a forward graph approximates a spec graph,
- a backward graph approximates the expected adjoint computation,
- a normal-form operator such as convolution or softmax-axis preserves an approximation relation,
- an FP32/CROWN bridge carries finite arithmetic assumptions into a bound statement.

The `NF` rounded-real bounds for sigmoid, logistic, and mean are built on `divPosErrorBound`,
with `sigmoid_bound_scalar_le_one` and `mean_row_bound_of_exact` as regression theorems. The FP32
MLP and CROWN theorems are named `approxTensor_reluTwoLayerMlp_fp32` and
`ibpBound_contains_reluTwoLayerMlp_fp32`; the `_fp32` suffix records that they are about the
rounded-real `FP32 := NF ...` model, not Lean's `Float32`.

CUDA, libtorch, and other native paths remain external unless a theorem explicitly connects the
native behavior to one of these approximation relations.

## Verification Proofs

Checker code lives primarily in `NN/Verification`; theorem-level soundness for bound propagation and
certificate families often lives in `NN/MLTheory`. This directory contains proof pieces that support
those paths, such as ODE enclosure facts and runtime-approximation bridges.

The intended pattern is:

1. a checker accepts a small artifact or graph condition,
2. the checker exposes the Lean predicate it recomputed,
3. a theorem states what that predicate implies over the relevant semantics,
4. remaining producer, parser, runtime, or floating-point assumptions are named.

These are different kinds of evidence. An executable regression run exercises a command path. A
checker says a finite artifact satisfied a Lean side predicate. A theorem says the predicate implies
a mathematical statement under stated hypotheses. Good TorchLean examples should make clear which
kind of evidence they are presenting and where the stronger statement lives.

## Tensor And Linear Algebra Proofs

The tensor proofs provide the quiet infrastructure used everywhere else: pointwise algebra, folds,
norm/bound facts, and finite matrix facts. `Tensor/Euclidean.lean` equips `Rep ℝ s` with an
`InnerProductSpace ℝ` instance, and the norm lemmas in `Basic/` are derived from it. The factorization files contain reconstruction and
orthonormality facts used by optimizer and linear-algebra developments, including Muon-style
orthogonalization certificates in `NN/MLTheory/Optimization`.

## RL Proofs

The RL proof files name the pieces that are easy to blur in executable examples:

- a Lean-native environment versus a Gymnasium subprocess,
- transition records versus external observations,
- replay-buffer structural invariants,
- MDP and finite stochastic MDP facts,
- checked-runtime bridges for floating-point rollouts.

Gymnasium is an external environment boundary. TorchLean proof statements are about the boundary
object once an external observation has been parsed, checked, and admitted into the TorchLean side.

## Adding Proofs

Add a proof here when it is reusable across examples or checkers. Keep one-off command tests in
`NN/Tests` and runnable examples in `NN/Examples`. If a theorem belongs to a specific theory family,
such as CROWN, optimizer laws, or learning theory, it may belong under `NN/MLTheory` instead.

Document the following information in each new proof file:

- what semantic object is being proved about,
- whether the scalar world is exact `ℝ`, executable IEEE-style, `Float`, or another scalar model,
- what runtime or external assumptions remain,
- which example or checker exercises the theorem, if one exists.

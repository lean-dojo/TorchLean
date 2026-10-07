# `NN/Proofs`

This directory contains theorems about TorchLean's tensors, autograd, graph translations,
numerical bounds, and verification checkers.

The curated import is:

```lean
import NN.Proofs
```

Ordinary model code should not import this directory. Use it when the file is proving a fact about
the spec layer, the runtime approximation layer, a graph translation, a floating-point model, or a
verification checker.

## Folder Map

| Folder | What it proves |
| --- | --- |
| `Tensor/` | Tensor algebra, folds, bounds/norms, finite linear algebra, and factorization facts. |
| `Autograd/` | Selected reverse-mode/autograd correctness facts, Fréchet derivative rules, tape algebra, runtime links, and training-step algebra. |
| `RuntimeApprox/` | Approximation relations between spec-level operations, graph forward/backward computations, normal-form operator rules, convolution, softmax-axis, FP32/CROWN bridges, and scale/tolerance lemmas. |
| `Backend/` | Planner dispatch, attention selection, lossless grouping, numerical-policy guards. |
| `Models/` | Attention masks, weights, permutation equivariance, and closed rational MLP examples. |
| `Analysis/` | Analytic facts for softmax, normalization, dropout, FFT, Lipschitz-style statements, and related helper theory. |
| `Gradients/` | Smaller gradient facts for layers and activations. |
| `RL/` | MDP, environment, replay-buffer, Gymnasium-boundary, DQN/PPO-adjacent, and checked-runtime RL facts. |
| `Verification/ODE/` | ODE enclosure and corridor facts used by learned sub/supersolution checkers. |
| `Probability/` | Probability and diffusion-forward helper facts. |
| `Utils/` | List and real-function helper lemmas shared by the other folders. |

## Autograd Proofs

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

Runtime approximation theorems bound forward and backward error using explicit local operator
assumptions and tolerances.

The `NF` rounded-real bounds for sigmoid, logistic, and mean are built on `divPosErrorBound`,
with `reciprocal_sigmoid_bound_scalar_le_one` and `mean_row_bound_of_exact` as regression theorems.
The FP32 MLP and CROWN theorems are named `approxTensor_reluTwoLayerMlp` and
`ibpBound_contains_reluTwoLayerMlp`; their `FP32` namespace identifies the
rounded-real `FP32 := NF ...` model, distinct from Lean's `Float32`.

CUDA, libtorch, and other native paths remain external unless a theorem explicitly connects the
native behavior to one of these approximation relations.

## Verification Proofs

Checker code lives primarily in `NN/Verification`; theorem-level soundness for bound propagation and
certificate families often lives in `NN/MLTheory`. This directory contains proof pieces that support
those paths, such as ODE enclosure facts and runtime-approximation bridges.

See [trust boundaries](../../docs/TRUST_BOUNDARIES.md) for the distinction between checker
acceptance, soundness theorems, and native execution evidence.

## Tensor And Linear Algebra Proofs

`Tensor/Euclidean.lean` equips `Rep ℝ s` with an `InnerProductSpace ℝ` instance, and the norm
lemmas in `Basic/` are derived from it. The factorization files contain reconstruction and
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
